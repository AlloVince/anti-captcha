<?php
require_once('vendor/autoload.php');

use Gregwar\Captcha\CaptchaBuilder;

function microtime_float()
{
    list($usec, $sec) = explode(" ", microtime());
    return ((float)$usec + (float)$sec);
}

while(true) {
    $start = microtime_float();
    $builder = new CaptchaBuilder;
    $builder->build(200, 80);
    $text = $builder->getPhrase();
    $builder->save("samples/$text.jpg");
    $end = microtime_float();
    $diff = ($end - $start) * 1000;
    echo "Generated samples/$text.jpg in $diff ms\n";
}

